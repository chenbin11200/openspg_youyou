# coding: utf-8
# Copyright 2023 OpenSPG Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.


"""
    knext

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Generated by: https://openapi-generator.tech
"""


import pprint
import re  # noqa: F401

import six

from knext.rest.configuration import Configuration


class MappingFilter(object):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    """
    Attributes:
      openapi_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    openapi_types = {"column_name": "str", "column_value": "str"}

    attribute_map = {"column_name": "columnName", "column_value": "columnValue"}

    def __init__(
        self, column_name=None, column_value=None, local_vars_configuration=None
    ):  # noqa: E501
        """MappingFilter - a model defined in OpenAPI"""  # noqa: E501
        if local_vars_configuration is None:
            local_vars_configuration = Configuration()
        self.local_vars_configuration = local_vars_configuration

        self._column_name = None
        self._column_value = None
        self.discriminator = None

        if column_name is not None:
            self.column_name = column_name
        if column_value is not None:
            self.column_value = column_value

    @property
    def column_name(self):
        """Gets the column_name of this MappingFilter.  # noqa: E501


        :return: The column_name of this MappingFilter.  # noqa: E501
        :rtype: str
        """
        return self._column_name

    @column_name.setter
    def column_name(self, column_name):
        """Sets the column_name of this MappingFilter.


        :param column_name: The column_name of this MappingFilter.  # noqa: E501
        :type: str
        """

        self._column_name = column_name

    @property
    def column_value(self):
        """Gets the column_value of this MappingFilter.  # noqa: E501


        :return: The column_value of this MappingFilter.  # noqa: E501
        :rtype: str
        """
        return self._column_value

    @column_value.setter
    def column_value(self, column_value):
        """Sets the column_value of this MappingFilter.


        :param column_value: The column_value of this MappingFilter.  # noqa: E501
        :type: str
        """

        self._column_value = column_value

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.openapi_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(
                    map(lambda x: x.to_dict() if hasattr(x, "to_dict") else x, value)
                )
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(
                    map(
                        lambda item: (item[0], item[1].to_dict())
                        if hasattr(item[1], "to_dict")
                        else item,
                        value.items(),
                    )
                )
            else:
                result[attr] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, MappingFilter):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, MappingFilter):
            return True

        return self.to_dict() != other.to_dict()